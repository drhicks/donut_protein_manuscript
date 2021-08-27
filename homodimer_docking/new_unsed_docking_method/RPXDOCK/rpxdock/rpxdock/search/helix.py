import logging, numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search

log = logging.getLogger(__name__)

def helix_get_sample_hierarchy(body, hscore, extent=100):
   "set up XformHier with appropriate bounds and resolution"
   cart_xhresl, ori_xhresl = hscore.base.attr.xhresl
   rg = body.rg()
   cart_samp_resl = 0.707 * cart_xhresl
   ori_samp_resl = cart_samp_resl / rg * 180 / np.pi
   # print(cart_samp_resl, rg, ori_samp_resl)
   ori_samp_resl = min(ori_samp_resl, ori_xhresl)
   # print(f"helix_get_sample_hierarchy cart: {cart_samp_resl} ori: {ori_samp_resl}")
   ncart = np.ceil(extent * 2 / cart_samp_resl)
   cartlb = np.array([-extent] * 3)
   cartub = np.array([extent] * 3)
   cartbs = np.array([ncart] * 3, dtype="i")
   xh = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, ori_samp_resl)
   assert xh.sanity_check(), "bad xform hierarchy"
   log.info(
      f"XformHier {xh.size(0):,} {xh.cart_bs} {xh.ori_resl} {xh.cart_lb} {xh.cart_ub}, body.copy()"
   )
   return xh

def make_helix(body, hscore, sampler, search=hier_search, **kw):
   kw = rp.Bunch(kw)
   kw.nresl = hscore.actual_nresl if kw.nresl is None else kw.nresl
   kw.output_prefix = kw.output_prefix if kw.output_prefix else sym
   t = rp.Timer().start()
   assert sampler is not None, 'sampler is required'

   evaluator = HelixEvaluator(body, hscore, **kw)
   xforms, scores, extra, stats = search(sampler, evaluator, **kw)

   ncat_pri = np.max(extra.helix_n_to_primary) + 1
   ncat_sec = np.max(extra.helix_n_to_secondry) + 1
   cat = extra.helix_n_to_primary
   cat += extra.helix_n_to_secondry * ncat_pri
   cat += extra.helix_cyclic * ncat_pri * ncat_sec

   ibest = rp.filter_redundancy(xforms, body, scores, categories=cat, **kw)

   if kw.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = kw.wts.sub(rpx=1, ncontact=0)
   wnct = kw.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, kw.nresl - 1, wrpx)
   ncontact, _ = evaluator(xforms, kw.nresl - 1, wnct)

   return rp.Result(
      body_=None if kw.dont_store_body_in_results else [body, body.copy()],
      attrs=dict(arg=kw, stats=stats, ttotal=t.total),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
      reslb=(["model"], extra.reslb),
      resub=(["model"], extra.resub),
      helix_n_to_primary=(["model"], extra.helix_n_to_primary),
      helix_n_to_secondry=(["model"], extra.helix_n_to_secondry),
      helix_cyclic=(["model"], extra.helix_cyclic),
      sym=(["model"], np.array(['H' + str(i) for i in extra.helix_cyclic])),
   )

class HelixEvaluator:
   def __init__(self, body, hscore, **kw):
      self.kw = rp.Bunch(kw)
      self.body = body.copy()
      self.hscore = hscore

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      kw = self.kw.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      body = self.body.copy()
      body2 = self.body.copy()
      xforms = xforms.reshape(-1, 4, 4)
      cart_extent = self.hscore.cart_extent[iresl]
      ori_extent = self.hscore.ori_extent[iresl]
      if not kw.helix_max_delta_z:
         helix_max_delta_z = body.radius_max() * 2 / kw.helix_min_isecond
      else:
         helix_max_delta_z = kw.helix_max_delta_z
      helix_max_iclash = int(kw.helix_max_isecond * 1.2 + 3)

      assert kw.symframe_num_helix_repeats >= kw.helix_max_isecond

      axis, ang = hm.axis_angle_of(xforms)
      ang = ang * 180 / np.pi

      aok = np.logical_and(ang >= kw.helix_min_primary_angle - ori_extent,
                           ang <= kw.helix_max_primary_angle + ori_extent)
      dhelix = np.abs(hm.hdot(axis, xforms[:, :, 3]))
      dok = np.logical_and(dhelix >= kw.helix_min_delta_z - cart_extent,
                           dhelix <= helix_max_delta_z + cart_extent)
      ok = np.logical_and(dok, aok)
      # ok = np.tile(True, len(xforms))

      scores = np.zeros((len(xforms), 2))
      ok[ok] = body.clash_ok(body2, xforms[ok], xeye, **kw)
      scores[ok, 0] = self.hscore.scorepos(body, body, xforms[ok], xeye, iresl, **kw)

      ok[ok] &= scores[ok, 0] >= kw.helix_min_primary_score
      ok[ok] &= scores[ok, 0] <= kw.helix_max_primary_score

      which2 = None
      # only check for 2ndary interaction when resolution is at least kw.helix_iresl_second_shift
      if iresl < kw.helix_iresl_second_shift:
         scores = scores[:, 0]
      else:
         xforms2 = xforms
         for i in range(2, helix_max_iclash):
            xforms2 = xforms @ xforms2
            ok[ok] = body.clash_ok(body2, xforms2[ok], xeye, **kw)
         xforms2 = xforms
         scores2 = np.zeros((kw.helix_max_isecond, len(xforms)))
         for i2 in range(2, kw.helix_max_isecond):
            xforms2 = xforms @ xforms2
            if i2 < kw.helix_min_isecond: continue
            scores2[i2, ok] = self.hscore.scorepos(
               body,
               body2,
               xforms2[ok],
               xeye,
               iresl - kw.helix_iresl_second_shift,
               **kw,
            )

         # xforms2 = xforms
         # scores2 = np.zeros((kw.helix_max_isecond, len(xforms)))
         # for i2 in range(2, kw.helix_max_isecond):
         #    xforms2 = xforms @ xforms2
         #    ok[ok] = body.clash_ok(body, xforms2[ok], xeye, **kw)  # it was you!
         #    scores2[i2, ok] = self.hscore.scorepos(body, body, xforms2[ok], xeye, iresl, **kw)
         # for i in range(kw.helix_max_isecond, helix_max_iclash):
         #    xforms2 = xforms @ xforms2
         #    ok[ok] = body.clash_ok(body, xforms2[ok], xeye, **kw)

         scores2[:, ~ok] = 0
         which2 = np.argmax(scores2, axis=0).astype('i8')
         scores[:, 1] = scores2[which2, range(len(xforms))]

         # summary depends on iresl stage, final is min score like before
         minsc = np.min(scores, axis=1)
         maxsc = np.max(scores, axis=1)
         mix0 = (iresl - kw.helix_iresl_second_shift + 1)
         mix = 0.5**mix0
         # print(iresl, mix0, mix)
         assert 0 <= mix <= 1
         scores = minsc * (1 - mix) + maxsc * mix

         # scores = np.min(scores, axis=1)
         # scores = scores[:, 0]

      helix_cyclic = np.repeat(1, len(scores))

      return scores, rp.Bunch(
         helix_n_to_primary=np.repeat(1, len(scores)),
         helix_n_to_secondry=which2,
         reslb=np.tile(0, len(scores)),
         resub=np.tile(len(body), len(scores)),
         helix_cyclic=helix_cyclic,
      )
